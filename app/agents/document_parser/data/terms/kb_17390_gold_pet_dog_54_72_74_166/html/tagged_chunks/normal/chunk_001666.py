from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,<br>보호자나 환자의 진술, 감정의의 추정 혹은 인정, 한국표준화가 이<br>루어지지 않고 신빙성이 적은 검사들(뇌 SPECT '
 '등)은 객관적 근거로<br>인정하지 않는다.<br>타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여 보상한다.<br>파) 외상후 '
 '스트레스장애, 우울증(반응성) 등의 질환, 정신분열증(조현<br>병), 편집증, 조울증(양극성장애), 불안장애, 전환장애, '
 "공포장애,<br>강박장애 등 각종 신경증 및 각종 인격장애는 보상의 대상이 되지<br>않는다.</p><p id='8'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001666',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
