from langchain_core.documents import Document

chunk = Document(
    page_content=("소멸)</h1><br><p id='55' data-category='paragraph' style='font-size:16px'>이 "
 '특별약관에서 정한 보상하는 손해가 더 이상 발생할 수<br>없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우<br>회사는「보험료 '
 '및 해약환급금 산출방법서」에서 정한 이<br>특별약관의 그 때까지 적립한 계약자적립액 및 미경과보험<br>료를 지급합니다.</p><p '
 "id='56' data-category='paragraph'"),
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
