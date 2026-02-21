from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 신경계․정신행동 장해의 경우 ① 개호(장해로 혼<br>자서 활동이 어려운 사람을 곁에서 돌보는 것) 여부 ② 객관적 이유 및 '
 "개호<br>의 내용을 추가로 기재하여야 한다.</p><h1 id='167' style='font-size:14px'>\uf000 "
 "장해분류별 판정기준</h1><h1 id='168' style='font-size:14px'>1. 눈의 장해</h1><h1 id='169' "
 "style='font-size:14px'>가"),
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
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_001477',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
