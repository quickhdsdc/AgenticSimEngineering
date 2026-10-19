<?php
// Local operator reset. Full PostgreSQL backup is required by the Python launcher.
require '/var/www/html/vendor/autoload.php';
$app = require '/var/www/html/bootstrap/app.php';
$app->make(Illuminate\Contracts\Console\Kernel::class)->bootstrap();
use Illuminate\Support\Facades\DB;
use App\Models\{WorkOrder, MaintenanceEvent, Line, ProductType, Shift, ScheduleChangeLog};

$result = DB::transaction(function () {
    if (WorkOrder::where('order_no', 'not like', 'A4S-%')->exists()) {
        throw new RuntimeException('Non-A4S orders exist. Review local reset scope first.');
    }
    $lines = Line::whereIn('code', ['A4S-FESTO', 'A4S-BACKUP'])->get()->keyBy('code');
    if ($lines->count() !== 2) throw new RuntimeException('Expected the two local A4S lines');
    $ids = $lines->pluck('id')->all();
    $orders = WorkOrder::where('order_no', 'like', 'A4S-%')->get();
    if ($orders->contains(function ($order) {
        return $order->status === 'IN_PROGRESS' || (float) $order->produced_qty > 0;
    })) throw new RuntimeException('An A4S order is active or has produced quantity; reset refused.');
    ScheduleChangeLog::whereIn('work_order_id', $orders->pluck('id'))->delete();
    foreach ($orders as $order) $order->delete();
    $maintenance = MaintenanceEvent::whereIn('line_id', $ids)->get();
    foreach ($maintenance as $event) $event->delete();
    $lines['A4S-FESTO']->update(['name' => 'Festo CP Lab · Housing Cell 1']);
    $lines['A4S-BACKUP']->update(['name' => 'Housing Cell 2']);
    $product = ProductType::where('code', 'A4S-DEMO-PART')->firstOrFail();
    $product->update(['name' => 'Phone housing · aluminium']);
    $product->lines()->syncWithoutDetaching($ids);
    foreach ($ids as $line) {
        foreach ([['D', '06:00', '18:00'], ['N', '18:00', '06:00']] as [$code, $start, $end]) {
            Shift::updateOrCreate(['code' => "A4S-$line-$code"], [
                'name' => "Housing $code", 'start_time' => $start, 'end_time' => $end,
                'days_of_week' => [1,2,3,4,5,6,7], 'line_id' => $line, 'is_active' => true,
            ]);
        }
    }
    return ['archived_orders' => $orders->count(), 'archived_maintenance' => $maintenance->count(),
            'line_id' => $lines['A4S-FESTO']->id, 'backup_line_id' => $lines['A4S-BACKUP']->id,
            'product_id' => $product->id];
});
echo 'A4S_RESET=' . json_encode($result) . PHP_EOL;
