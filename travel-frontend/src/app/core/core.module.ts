import { NgModule, Optional, SkipSelf } from '@angular/core';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { AppHttpInterceptor } from './interceptors/http.interceptor';

@NgModule({
  imports: [HttpClientModule],
  providers: [
    {
      provide:  HTTP_INTERCEPTORS,
      useClass: AppHttpInterceptor,
      multi:    true,
    },
  ],
})
export class CoreModule {
  constructor(@Optional() @SkipSelf() parent: CoreModule) {
    if (parent) {
      throw new Error('CoreModule has already been loaded. Import it only in AppModule.');
    }
  }
}
