from langchain_core.documents import Document

chunk = Document(
    page_content=('- 있으며 이 때의 보험료는 "보험료 및 해약환급금 산출방법서"에 따라 산출합니\n'
 '- 다.\n'
 '- \uf000 제5항에 따라 보험계약이 연장된 경우 계약자는 그 최초연장된 날로부터 90일 이\n'
 '- 내에 그 계약을 취소할 수 있으며, 계약자가 연장된 보험계약을 취소하는 경우 회\n'
 '- 사는 최초연장된 날 이후 계약자가 납입한 보험료 전액을 환급합니다.\n'
 '- \uf000 제5항에 따라 보험계약이 연장된 경우 보험계약의 연장일은 회사가 계약자의 재\n'
 '- 가입의사를 확인한 날(계약자 등이 회사에 보험금을 청구함으로써 계약자에게 연'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000538',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
