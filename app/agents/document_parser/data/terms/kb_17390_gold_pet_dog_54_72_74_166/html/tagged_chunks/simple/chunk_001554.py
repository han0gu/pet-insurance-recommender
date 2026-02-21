from langchain_core.documents import Document

chunk = Document(
    page_content=('중 어느 하나에 해당하는 경우를 말한다.<br>가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 3개의 척추체(척추뼈 몸 '
 "특별</p><br><p id='41' data-category='paragraph' style='font-size:14px'>통)를 "
 '유합(아물어 붙음) 또는 고정한 상태 약<br>나) 머리뼈(두개골)와 제1경추 또는 제1경추와 제2경추를 유합 또는 고 관<br>정한 '
 '상태<br>다) 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경추) 사이에 CT 검사<br>상, 두개 대후두공의 기저점(basion)과 '
 '축추 치돌기'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_001554',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
