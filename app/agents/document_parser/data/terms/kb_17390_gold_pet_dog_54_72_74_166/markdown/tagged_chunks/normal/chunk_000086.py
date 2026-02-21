from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서면(등기우편 등)으로 다시 알려드립니다.\n'
 '- \uf000 제1항 제2호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에는\n'
 '- 제15조(상해보험계약 후 알릴 의무) 제4항 또는 제5항에 따라 보험금을 지급합니\n'
 '- 다.\n'
 '- \uf000 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금 지급사유 발생에 영향을\n'
 '- 미쳤음을 회사가 증명하지 못한 경우에는 제4항 및 제5항에 관계없이 약정한 보험\n'
 '- 금을 지급합니다.\n'
 '- \uf000 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위반을 이유로 계약을 해지'),
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
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
