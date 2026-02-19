from langchain_core.documents import Document

chunk = Document(
    page_content=('환급되는 금액이 납입보험료를 초과하지 아니하는 보험으로서 보험계약 또는 보험료납 입영수증에 보험료 공제대상임이 표시된 보험의 보험료를 '
 '말한다.\n'
 '2. 모든 피보험자 또는 모든 보험수익자가「소득세법 시행령 제107조(장애인의 범위) 제 1항」에서 규정한 장애인인 보험\n'
 '【「소득세법 시행령 제107조(장애인의 범위) 제1항」에서 규정한 장애인】\n'
 '① 법 제51조 제1항 제2호에 따른 장애인은 다음 각 호의 어느 하나에 해당하는 자로 한다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 45},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000241',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
