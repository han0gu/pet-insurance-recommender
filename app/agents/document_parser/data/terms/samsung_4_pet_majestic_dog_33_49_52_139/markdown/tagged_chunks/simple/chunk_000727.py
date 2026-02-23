from langchain_core.documents import Document

chunk = Document(
    page_content=('- 정신체부위 또는 특정질병의 상태에 따라 「1개월부터 5년」또는「보험계약의 보험기\n'
 '- 간 전체」로 하며, 그 판단기준은 회사에서 정한 계약사정기준을 따릅니다. 다만, 개\n'
 '- 개인의 질병의 상태 등에 대한 의사의 소견에 따라서 다르게 적용할 수 있습니다.\n'
 '- ③ 제2항에도 불구하고 보험업법 제97조 제1항 제5호 및 동법 시행령 제43조의2 제1항\n'
 '- 에 따른 보장내용 등이 비슷한 보험계약(이하 「유사계약」이라 합니다)이 계약 청약\n'
 '- 일 현재 유지중이거나, 계약 청약일 전 6개월 이내에 계약자 및 피보험자의 요구 또는'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000727',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
