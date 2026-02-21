from langchain_core.documents import Document

chunk = Document(
    page_content=('. "동물병원"이란 동물진료업을 하는 장소로서 제17조에 따른 신고를 한 진료<br>특<br>기관을 말한다.<br>약</p><br><p '
 "id='32' data-category='paragraph' style='font-size:14px'>반</p><br><p id='33' "
 "data-category='paragraph' style='font-size:20px'>려동</p><p id='34' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 '펫보험(강아지)(무배당)(26.01)'),
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
 'indexing': {'chunk_id': 'chunk_001061',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
