from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 계약자 또는 피보험자가 지출한 아래의 비용가. 피보험자가 제11조(손해방지의무)의 제1항 제1호의 손해의 방지 또는 경감을 위\n'
 '하여 지출한 필요 또는 유익하였던 비용\n'
 '나. 피보험자가 제11조(손해방지의무)의 제1항 제2호의 제3자로부터 손해의 배상을\n'
 '받을 수 있는 그 권리를 지키거나 행사하기 위하여 지출한 필요 또는 유익하였\n'
 '던 비용- 22 -【설명】제3자로부터 손해의 배상을 받을 수 있는 경우에 피보험자가 손해배상청구를 위\n'
 '해 내용증명, 재산조사, 강제집행 등을 수행하고자 지출한 각종 비용을 의미합니'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000120',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
