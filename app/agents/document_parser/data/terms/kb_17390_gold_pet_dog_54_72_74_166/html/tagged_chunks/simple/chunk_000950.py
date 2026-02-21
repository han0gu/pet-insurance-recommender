from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약일은 제1회 보험료를 받은 날로 합니다.<br>\uf000 제1항 내지 제4항에도 불구하고 이 특별약관의 보험계약일로부터 그날을 '
 '포함하<br>여 1년 이내에 발생한 슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성부전<br>또는 기타 이들과 유사한 사고에 대해서는 '
 "보험금을 지급하지 않습니다.</p><br><h1 id='138' style='font-size:14px'>예 시 1</h1><br><h1 "
 "id='139' style='font-size:14px'>반려동물의료비의 보장개시일</h1><br><figure"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000950',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
