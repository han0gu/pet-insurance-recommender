from langchain_core.documents import Document

chunk = Document(
    page_content=('나) 음경의 1/2 이상이 결손되었거나 질구 협착으로 성생활이 불가능한 때\n'
 '다) 폐질환 또는 폐 부분절제술 후 일상생활에서 호흡곤란으로 지속적\n'
 '인 산소치료가 필요하며, 폐기능 검사(PFT)상 폐환기 기능(1초간- \n'
 '노력성 호기량, FEV1)이 정상예측치의 40% 이하로 저하된 때\n'
 '6) 흉복부, 비뇨생식기계 장해는 질병 또는 외상의 직접 결과로 인한 장해\n'
 '를 말하며, 노화에 의한 기능장해 또는 질병이나 외상이 없는 상태에서\n'
 '예방적으로 장기를 절제, 적출한 경우는 장해로 보지 않는다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000928',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
