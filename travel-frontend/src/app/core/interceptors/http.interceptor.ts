import { Injectable } from '@angular/core';
import {
  HttpEvent, HttpHandler, HttpInterceptor, HttpRequest, HttpErrorResponse,
} from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { SessionService } from '../services/session.service';

export interface ApiError {
  status:  number;
  message: string;
}

@Injectable()
export class AppHttpInterceptor implements HttpInterceptor {

  constructor(private session: SessionService) {}

  intercept(req: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    const cloned = req.clone({
      setHeaders: { 'X-Session-Id': this.session.getSessionId() },
    });

    return next.handle(cloned).pipe(
      catchError((err: HttpErrorResponse) => {
        const msg = err.status === 0
          ? 'Cannot reach server. Is the backend running on port 8000?'
          : `Server error ${err.status}: ${err.statusText}`;
        return throwError(() => ({ status: err.status, message: msg } as ApiError));
      }),
    );
  }
}
