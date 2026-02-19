from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 수의사는 직접 진료하거나 검안한 동물에 대한 진단서, 검안서, 증명서 또는 처방전의 발급을 요구받았을 때에는 정당한 사유 없이 '
 '이를 거부하여서는 아니된다. ④ 제1항부터 제3항까지의 규정에 따른 진단서, 검안서, 증명서 또는 처방전의 서식, 기재사항, 그 밖에 '
 '필요한 사항은 농림축산식품부령으로 정한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 120},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000773',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
