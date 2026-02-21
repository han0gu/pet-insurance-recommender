from langchain_core.documents import Document

chunk = Document(
    page_content=('2경추) 사이에 CT 검사<br>상, 두개 대후두공의 기저점(basion)과 축추 치돌기 상단사이의 거<br>리(BDI : '
 'Basion-Dental Interval)에 뚜렷한 이상전위가 있는 상태<br>별<br>라) 상위목뼈(상위경추: 제1, 2경추) CT '
 '검사상, 환추 전방 궁(arch)의<br>표<br>후방과 치상돌기의 전면과의 거리(ADI: Atlanto-Dental '
 "Interval)<br>에 뚜렷한 이상전위가 있는 상태</p><br><p id='42' "
 "data-category='list'></p><br><h1 id='43'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001555',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
