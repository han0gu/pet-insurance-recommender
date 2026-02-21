from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자 또는 피보험자(법인인 경우에는 그 이사 또는 법인의 업무를 집행하는 그 밖의<br>기관)또는 이들의 법정대리인의 '
 '고의<br>2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태<br>3. 지진, 분화, 홍수, '
 '해일 또는 이와 비슷한 천재지변<br>4'),
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
