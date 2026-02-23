from langchain_core.documents import Document

chunk = Document(
    page_content=('. 예를 들어, 보<br>험계약일이 2023년 4월 1일인 경우 보험연도는 4월 1일<br>부터 차년도 3월 31일까지 1년을 '
 "말합니다.</p><h1 id='20' style='font-size:20px'>【중도인출금의 한도 예시】</h1><br><p "
 "id='21' data-category='paragraph' style='font-size:16px'>계약자가 요청한 시점에서 계산된 "
 '기본계약 해약환급금<br>과 기본계약 적립부분 해약환급금 중 적은 금액이 100만<br>원인 경우 중도인출 가능액은 80만원(100만원의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
