from langchain_core.documents import Document

chunk = Document(
    page_content=('- 적인 간헐적 인공요도가 필요한 때\n'
 '- 나) 음경의 1/2 이상이 결손되었거나 질구 협착으로 성생활이 불가능한 때\n'
 '- 다) 폐질환 또는 폐 부분절제술 후 일상생활에서 호흡곤란으로 지속적인 산소\n'
 '- 치료가 필요하며, 폐기능 검사(PFT)상 폐환기 기능(1초간 노력성 호기량,\n'
 '- FEV1)이 정상예측치의 40% 이하로 저하된 때\n'
 '- 6) 흉복부, 비뇨생식기계 장해는 질병 또는 외상의 직접 결과로 인한 장해를 말하\n'
 '- 며, 노화에 의한 기능장해 또는 질병이나 외상이 없는 상태에서 예방적으로 장'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000829',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
