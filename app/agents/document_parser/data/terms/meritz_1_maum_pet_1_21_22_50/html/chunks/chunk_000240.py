from langchain_core.documents import Document

chunk = Document(
    page_content=("id='66' style='font-size:14px'>- 25 -</footer><p id='67' "
 "data-category='paragraph' style='font-size:14px'>차감한 금액을 손해액으로 간주하여 제1항에 의한 "
 "보상할 금액을 결정합니다.</p><br><p id='68' data-category='paragraph' "
 "style='font-size:14px'>③ 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 "
 '따른<br>지급보험금 결정에는 영향을 미치지 않습니다.</p><h1'),
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
