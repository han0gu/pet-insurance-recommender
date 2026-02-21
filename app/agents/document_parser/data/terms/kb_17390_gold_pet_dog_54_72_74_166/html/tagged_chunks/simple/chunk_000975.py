from langchain_core.documents import Document

chunk = Document(
    page_content=('. 첩모난생(속눈썹 질환) 및 눈물샘 치료(누루관시술 등) 등의 안검 외·내반 및<br>비루관 관련 질환으로 인한 '
 "비용<br>특</p><br><h1 id='164' style='font-size:16px'>\uf000 제2항에서 정한 조치에 다른 "
 "진료를 병행하여 실시한 경우에는 제2항에서 정한 조</h1><br><p id='165' data-category='paragraph' "
 "style='font-size:16px'>치(마취 비용을 포함합니다.)에</p><br><p id='166' "
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
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000975',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
