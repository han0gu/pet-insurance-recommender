from langchain_core.documents import Document

chunk = Document(
    page_content=("전환)</h1><br><p id='67' data-category='list' style='font-size:14px'>① 회사는 이 "
 '특별약관이 부가된 전환계약을「소득세법 제59조의4(특별세액공제) 제1항<br>제1호」에 해당하는 장애인전용보험으로 전환하여 '
 '드립니다.<br>② 제1항에 따라 전환대상계약이 장애인전용보험으로 전환된 후부터 납입된 전환대상계약<br>보험료는 보험료 납입영수증에 '
 "장애인전용 보장성보험료로 표시됩니다.</p><h1 id='68' style='font-size:14px'>【예시】</h1><br><p"),
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
