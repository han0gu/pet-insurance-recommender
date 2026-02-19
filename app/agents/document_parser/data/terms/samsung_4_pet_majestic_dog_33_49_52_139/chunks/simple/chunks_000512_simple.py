from langchain_core.documents import Document

chunk = Document(
    page_content=('정한 수가코드를 확인할 수 있는 경우 회사는 제3조(창상봉합술의 정의와 장소)에서 정한 치료에 포함하여 보장합니다.\n'
 '<예시안내>\n'
 '자동차보험 또는 산재보험 등 「국민건강보험법」 또는 「의료급여법」 을 적용받지 못하는 사고로 인하여 창상봉합술 치료를 받은 경우에도 '
 '보장하는 수가코드를 확인할 수 있는 경우 회사는 창상 봉합술 치료비를 지급합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000512',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
