from langchain_core.documents import Document

chunk = Document(
    page_content=('- 그 계약을 취소할 수 있으며, 계약자가 연장된 보험계약을 취소하는 경우 회사는 최초\n'
 '- 연장된 날 이후 계약자가 납입한 보험료 전액을 환급합니다.\n'
 '- ⑦ 제5항에 따라 보험계약이 연장된 경우 보험계약의 연장일은 회사가 계약자의 재가입\n'
 '- 의사를 확인한 날(계약자 등이 회사에 보험금을 청구함으로써 계약자에게 연락이 닿\n'
 '- 아 회사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다. 회사는 계약자 등이\n'
 '- 회사에 보험금을 청구하는 등 계약자에게 연락이 닿으면 제3항의 내용과 90일 이내'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000387',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
