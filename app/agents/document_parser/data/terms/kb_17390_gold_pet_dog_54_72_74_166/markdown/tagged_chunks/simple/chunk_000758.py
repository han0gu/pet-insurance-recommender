from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이라 합니다)을 부가할 수 있습니다.\n'
 '- \uf000 이 특별약관의 보험기간은 보통약관의 보험기간이 끝나는 날의 12개월 이전까지로\n'
 '- 합니다.\n'
 '- \uf000 보통약관에 사망보험금을 지급하는 특별약관(이하 "사망보장특별약관"이라 합니\n'
 '- 다)이 부가되어 있는 경우에도 이 특별약관을 적용합니다.\n'
 '# 제2조(지급사유)# \uf000 회사는 특별약관의보험기간 중 의료법 제3조에 정한 국내의 종합병원 또는 이와동등하다고 회사가 인정하는 '
 '의료기관에서 전문의 자격증을 가진 자가 실시한 진'),
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
 'indexing': {'chunk_id': 'chunk_000758',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
