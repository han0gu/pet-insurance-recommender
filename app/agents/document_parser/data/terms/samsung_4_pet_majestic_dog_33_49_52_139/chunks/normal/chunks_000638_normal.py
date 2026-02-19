from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한, 보험계약의 연장은 기본계약의 보험기간 내에서만 가능합니다. ⑥ 제5항에 따라 보험계약이 연장된 경우 계약자는 그 최초연장된 '
 '날로부터 90일 이내에 그 계약을 취소할 수 있으며, 계약자가 연장된 보험계약을 취소하는 경우 회사는 최초 연장된 날 이후 계약자가 '
 '납입한 보험료 전액을 환급합니다. ⑦ 제5항에 따라 보험계약이 연장된 경우 보험계약의 연장일은 회사가 계약자의 재가입 의사를 확인한 '
 '날(계약자 등이 회사에 보험금을 청구함으로써 계약자에게 연락이 닿 아 회사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 108},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000638',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
