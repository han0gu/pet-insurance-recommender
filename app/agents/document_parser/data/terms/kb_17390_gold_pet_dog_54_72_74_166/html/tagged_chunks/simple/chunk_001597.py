from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 제거<br>가 불가능한 경우에는 고정물 등이 있는 상태에서 장해를 평가한다.<br>2) 관절을 사용하지 않아 발생한 일시적인 '
 '기능장해(예를 들면 캐스트로 환<br>부를 고정시켰기 때문에 치유 후의 관절에 기능장해가 발생한 경우)는<br>장해로 평가하지 '
 '않는다.<br>3) ‘다리’라 함은 엉덩이관절(고관절)부터 발목관절(족관절)까지를 말한다.<br>4) ‘다리의 3대 관절’이라 함은 '
 "엉덩이관절(고관절), 무릎관절(슬관절),<br>발목관절(족관절)을 말한다.</p><p id='105'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001597',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
