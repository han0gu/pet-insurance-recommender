from langchain_core.documents import Document

chunk = Document(
    page_content=('- ·반려동물의료비보험금 : 15만원\n'
 '- ·지급금액 = {(218만원 – 15만원 - 3만원) x 70%, 100만원} 중 적은 금액\n'
 '- = 100만원\n'
 '\uf000 제1항에서 "치료구분별 대상원인"이란 아래의 상해 또는 질병을 말합니다.| 치 료 구 분 | 치 료 구 분 | 대 상 원 인 '
 '|\n'
 '| --- | --- | --- |\n'
 '| MRI/CT | MRI/CT | 상해 또는 질병 |\n'
 '| 백내장/녹내장수술 | 백내장/녹내장수술 | 백내장 또는 녹내장 |\n'
 '| 특정처치 (이물제거) | 이물제거(내시경) | 이물섭취 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000578',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
