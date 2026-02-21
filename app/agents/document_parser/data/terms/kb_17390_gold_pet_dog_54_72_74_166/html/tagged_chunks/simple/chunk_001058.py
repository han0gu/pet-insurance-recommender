from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류 병</p><br><p id='28' "
 "data-category='paragraph' style='font-size:16px'>\uf000 제1항 제4호의 사고증명서는 수의사법 "
 "제2조(정의)에서 규정한 동물병원에서 수의<br>사가 발급한 것이어야 합니다.</p><br><h1 id='29' "
 "style='font-size:16px'>관 련 법 규 수의사법 제2조(정의)</h1><br><h1 id='30' "
 "style='font-size:16px'>이 법에서 사용하는 용어의 뜻은 다음과"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001058',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
