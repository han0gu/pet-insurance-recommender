from langchain_core.documents import Document

chunk = Document(
    page_content=('되었을 때에는 장해분류표에서 정한 장해지급률을 이 특별약관의 보험가입금액에 곱\n'
 '질\n'
 '하여 산출한 금액을 일반상해후유장해(3~79%) 보험금으로 보험수익자에게 지급합니\n'
 '병\n'
 '다.제2조(보험금 지급에 관한 세부규정)- \uf000 제1조(보험금의 지급사유)에서 장해지급률이 상해 발생일부터 180일 이내에 확정\n'
 '- 상\n'
 '- 되지 않는 경우에는 상해 발생일부터 180일이 되는 날의 의사 진단에 기초하여 고\n'
 '- 해\n'
 '- 정될 것으로 인정되는 상태를 장해지급률로 결정합니다. 다만, 장해분류표에 장 및\n'
 '- 해판정시기를 별도로 정한 경우에는 그에 따릅니다. 질'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000251',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
