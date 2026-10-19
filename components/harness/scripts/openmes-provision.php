<?php
// Operator-only, executed inside the isolated a4s-openmes backend.
require '/var/www/html/vendor/autoload.php';
$app = require '/var/www/html/bootstrap/app.php';
$app->make(Illuminate\Contracts\Console\Kernel::class)->bootstrap();
$result = Illuminate\Support\Facades\DB::transaction(function () {
    $admin = App\Models\User::where('username', 'a4s-admin')->firstOrFail();
    if (App\Models\User::where('username', 'a4s-agent')->exists()) {
        throw new RuntimeException('Service identity exists; recover or rotate its token manually.');
    }
    $user = App\Models\User::create([
        'username' => 'a4s-agent', 'name' => 'A4S Harness',
        'email' => 'a4s-agent@example.invalid', 'password' => bin2hex(random_bytes(32)),
        'force_password_change' => false, 'tenant_id' => $admin->tenant_id,
    ]);
    $user->assignRole('Supervisor');
    $lines = [];
    foreach (['A4S-FESTO' => 'A4S Festo CP Lab', 'A4S-BACKUP' => 'A4S Backup (simulated)'] as $code => $name) {
        $lines[] = App\Models\Line::firstOrCreate(['code' => $code], [
            'name' => $name, 'is_active' => true, 'tenant_id' => $admin->tenant_id,
        ]);
    }
    $product = App\Models\ProductType::firstOrCreate(['code' => 'A4S-DEMO-PART'], [
        'name' => 'A4S Demo Part', 'is_active' => true, 'tenant_id' => $admin->tenant_id,
    ]);
    $product->lines()->syncWithoutDetaching(array_map(fn ($line) => $line->id, $lines));
    $user->lines()->syncWithoutDetaching(array_map(fn ($line) => $line->id, $lines));
    return [
        'token' => $user->createToken('a4s-harness-local', ['*'], now()->addDays(30))->plainTextToken,
        'token_expiry_days' => 30, 'product_type_id' => $product->id,
        'lines' => array_map(fn ($line) => ['id' => $line->id, 'code' => $line->code], $lines),
    ];
});
echo 'A4S_PROVISION=' . json_encode($result) . PHP_EOL;
