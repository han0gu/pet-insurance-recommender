from langchain_core.documents import Document

chunk = Document(
    page_content=('. 핵연료물질 또는 핵연료 물질에 의하여 오염된 물질의 방사성, 폭발성, 그 밖<br>회사는 그 증가된 손해를 보상하여 드리지 않으며, '
 '제1항 제3호의 통지를 게을리 성<br>의 유해한 특성 또는 이들의 특성에 의한 사고로 생긴 손해에 대한 배상책임<br>한 때에는 '
 '소송비용과 변호사비용도 보상하여 드리지 않습니다. 다만, 계약자 또 특<br>5'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001158',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
