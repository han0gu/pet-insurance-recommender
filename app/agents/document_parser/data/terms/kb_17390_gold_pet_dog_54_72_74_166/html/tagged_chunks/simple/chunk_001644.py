from langchain_core.documents import Document

chunk = Document(
    page_content=('음경의 1/2 이상이 결손되었거나 질구 협착으로 성생활이 불가능한 때<br>다) 폐질환 또는 폐 부분절제술 후 일상생활에서 호흡곤란으로 '
 "지속적<br>인 산소치료가 필요하며, 폐기능 검사(PFT)상 폐환기 기능(1초간</h1><br><p id='175' "
 "data-category='list'></p><br><p id='176' data-category='paragraph' "
 "style='font-size:16px'>노력성 호기량, FEV1)이 정상예측치의 40% 이하로 저하된 때<br>6) 흉복부, "
 '비뇨생식기계 장해는 질병 또는 외상의'),
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
 'indexing': {'chunk_id': 'chunk_001644',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
