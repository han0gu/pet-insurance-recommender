from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 이 경<br>우 동물병원에서 발급한 소견서를 제출하여야 합니다.<br>\uf000 제1항의 경우 무지개다리위로금보장개시일은 '
 "계약일로부터 그날을 포함하여 30일</p><br><table id='55' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>이 지난날의</td><td>다음날로 "
 '합니다. 단, 계약일은 제1회 보험료를 받은 날로 합니다.</td></tr><tr><td colspan="2">예 시 무지개다리위로금의 '
 '보장개시일 <figure><img alt="계약일 보장개시일'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001077',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
