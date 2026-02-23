from langchain_core.documents import Document

chunk = Document(
    page_content=('사유)에 해당하는 경우에 한하여 해당 보험금을 지급합니다.③ 피보험자가 응급실 내원하던 도중 사망하는 경우에도 「아나필락시스」 를 '
 '직접적인\n'
 '원인으로 사망한 것이 확인된 경우에는 응급실에 내원하여 진단받은 것으로 보아 제2\n'
 '항을 적용합니다.# 제3조 (아나필락시스의 정의 및 진단확정)① 이 특별약관에서 「아나필락시스」 이라 함은 한국표준질병·사인분류에 있어서 '
 '[별표-\n'
 '상해관련4]아나필락시스 분류표에서 정한 상병을 말합니다.\n'
 '② 「아나필락시스」 의 진단확정은 의료법 제3조(의료기관)에서 규정한 국내의 병원, 의'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000314',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
