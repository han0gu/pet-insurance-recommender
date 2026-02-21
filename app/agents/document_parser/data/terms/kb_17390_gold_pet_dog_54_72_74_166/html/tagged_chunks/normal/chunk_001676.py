from langchain_core.documents import Document

chunk = Document(
    page_content=('밀어 이동이 가능한 상태</td><td>보 통약 관</td></tr><tr><td>3) 목발 또는 보행기(walker)를 사용하지 않으면 '
 '독립적 20% 인 보행이 불가능한 상태</td><td>특별 약</td></tr><tr><td>4) 보조기구 없이 독립적인 보행은 가능하나 '
 '보행시 파행 (절뚝거림)이 있으며, 난간을 잡지 않고는 계단을 오 10% 르내리기가 불가능한 상태 또는 평지에서 100m 이상을 걷지 '
 '못하는 상태</td><td>관</td></tr><tr><td rowspan="4">음식물 섭취</td><td>1) 입으로'),
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
 'indexing': {'chunk_id': 'chunk_001676',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
