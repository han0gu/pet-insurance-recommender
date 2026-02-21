from langchain_core.documents import Document

chunk = Document(
    page_content=("id='88' style='font-size:14px'>- 39 -</footer><h1 id='89' "
 "style='font-size:18px'>단체계약 보험료정산 추가특별약관(Ⅱ)</h1><h1 id='90' "
 "style='font-size:14px'>제1조(보험료의 정산)</h1><br><p id='91' data-category='list' "
 "style='font-size:14px'>① 회사는 단체계약 특별약관(이하“특별약관”이라 합니다) 제4조(보험의 목적의 증가 "
 '감소<br>또는 교체) 제2항 및 보통약관 제16조(계약 후'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
