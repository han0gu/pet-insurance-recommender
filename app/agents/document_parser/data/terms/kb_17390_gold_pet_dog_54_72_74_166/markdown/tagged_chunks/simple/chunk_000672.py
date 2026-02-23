from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 핵연료물질 또는 핵연료 물질에 의하여 오염된 물질의 방사성, 폭발성, 그 밖\n'
 '회사는 그 증가된 손해를 보상하여 드리지 않으며, 제1항 제3호의 통지를 게을리 성\n'
 '의 유해한 특성 또는 이들의 특성에 의한 사고로 생긴 손해에 대한 배상책임\n'
 '한 때에는 소송비용과 변호사비용도 보상하여 드리지 않습니다. 다만, 계약자 또 특\n'
 '5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염으로 인한 손해\n'
 '는 피보험자가 상법 제657조 제1항에 의해 보험사고의 발생을 회사에 알린 경우 약'),
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
 'indexing': {'chunk_id': 'chunk_000672',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
