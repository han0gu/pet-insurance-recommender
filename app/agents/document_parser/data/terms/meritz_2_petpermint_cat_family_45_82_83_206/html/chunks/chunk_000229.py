from langchain_core.documents import Document

chunk = Document(
    page_content=("보험수익자에게 보험계약대출 사실을 통지할 수<br>있습니다.</p><h1 id='12' "
 "style='font-size:20px'>제37조(배당금의 지급)</h1><br><p id='13' "
 "data-category='paragraph' style='font-size:20px'>회사는 이 계약에 대하여 계약자에게 배당금을 "
 "지급하지 않<br>습니다.</p><h1 id='14' style='font-size:20px'>제38조(중도인출)</h1><br><p "
 "id='15' data-category='paragraph'"),
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
