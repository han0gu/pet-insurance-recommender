from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써【별표\n'
 '- 5】치아파절 분류표에서 정한 치아파절로 진단확정 된 경우 연간 3회에 한하여\n'
 '- 사고시마다 이 특별약관의 보험가입금액을 치아파절진단비로 보험수익자에게\n'
 '- 지급합니다.\n'
 '- \uf000 제1항에서 "연간"이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까\n'
 '- 지 기간을 의미합니다.\n'
 '# \uf000 제1항에서 치아파절진단비의 진단일자는 사고일을 기준으로 합니다.- 제2조(보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000318',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
