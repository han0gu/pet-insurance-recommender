from langchain_core.documents import Document

chunk = Document(
    page_content=("id='12' data-category='list'></p><br><h1 id='13' style='font-size:16px'>가) "
 "“뇌전증”이라 함은 돌발적 뇌파이상을 나타내는 뇌질환으로 발작</h1><br><p id='14' "
 "data-category='paragraph' style='font-size:16px'>(경련, 의식장해 등)을 반복하는 것을 "
 '말한다.<br>나) 뇌전증 발작의 빈도 및 양상은 지속적인 항뇌전증제(항경련제) 약물<br>로도 조절되지 않는 뇌전증을 말하며, '
 '진료기록에 기재되어 객관적<br>으로 확인되는'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001670',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
