from langchain_core.documents import Document

chunk = Document(
    page_content=('<소득세법 시행규칙 제61조의3 (공제대상보험료의 범위)>\n'
 '영 제118조의4제2항 각 호 외의 부분에서 "기획재정부령으로 정하는 것"이란 만기에 환급되는 금액이 납입보험료를 초과하지 아니하는 '
 '보험으로서 보험계약 또는 보험료납입영수증에 보험료 공제대상임이 표시된 보험의 보험료를 말한다.\n'
 '2. 모든 피보험자 또는 모든 보험수익자가 「소득세법 시행령 제107조(장애인의 범위) 제 1항」 에 서 규정한 장애인인 보험\n'
 '【용어해설】\n'
 'く 「소득세법 시행령 제107조(장애인의 범위)」 에서 규정한 장애인>'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 35},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000170',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
