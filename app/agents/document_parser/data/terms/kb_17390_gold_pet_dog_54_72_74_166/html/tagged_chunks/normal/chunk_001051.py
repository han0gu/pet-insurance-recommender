from langchain_core.documents import Document

chunk = Document(
    page_content=('. 첩모난생(속눈썹 질환) 및 눈물샘 치료(누루관시술 등) 등의 안검 외·내반 및<br>비루관 관련 질환으로 인한 비용<br>\uf000 '
 '제2항에서 정한 조치에 다른 진료를 병행하여 실시한 경우에는 제2항에서 정한 조<br>치(마취 비용을 포함합니다.)에 대한 보험금은 '
 "지급하지 않습니다.</p><br><p id='21' data-category='paragraph' "
 "style='font-size:16px'>제5조(보험금의 청구)</p><br><p id='22' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_001051',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
