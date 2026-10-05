package com.ergopower.wallet;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.content.SharedPreferences;
import android.text.InputType;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;

public class MainActivity extends Activity {
    private WebView webView;
    private SharedPreferences prefs;

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        prefs = getSharedPreferences("ergo_wallet", MODE_PRIVATE);
        webView = findViewById(R.id.webview);
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(false);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        webView.setWebViewClient(new WebViewClient());
        showServerDialog();
    }

    private void showServerDialog() {
        final EditText input = new EditText(this);
        input.setHint("https://your-wallet.up.railway.app");
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        input.setSingleLine(true);
        input.setText(prefs.getString("server_url", ""));

        new AlertDialog.Builder(this)
            .setTitle("ERGO Wallet Server")
            .setMessage("Enter the HTTPS address of your deployed ERGO Cash Token server.")
            .setView(input)
            .setCancelable(false)
            .setPositiveButton("Open Wallet", (dialog, which) -> {
                String url = input.getText().toString().trim();
                if (!url.startsWith("https://")) {
                    new AlertDialog.Builder(this)
                        .setTitle("HTTPS required")
                        .setMessage("For security, enter an address beginning with https://")
                        .setPositiveButton("Try Again", (d, w) -> showServerDialog())
                        .setCancelable(false)
                        .show();
                    return;
                }
                prefs.edit().putString("server_url", url).apply();
                webView.loadUrl(url);
            })
            .show();
    }

    @Override public void onBackPressed() {
        if (webView != null && webView.canGoBack()) webView.goBack(); else super.onBackPressed();
    }
}
