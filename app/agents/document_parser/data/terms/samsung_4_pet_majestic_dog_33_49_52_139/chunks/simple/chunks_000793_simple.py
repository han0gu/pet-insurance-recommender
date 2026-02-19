from langchain_core.documents import Document

chunk = Document(
    page_content=('Chart Title: <시안내>\n'
 'Chart Type: bar\n'
 '보상 | 보상제외\n'
 '퉈원없이 계속 입원 | -182년 | -182년\n'
 '치고된 최종 입원이 | -173년 | -172년\n'
 '7 제1항의 경우 피보험자가 보장개시일(책임개시일) 이후 입원하여 치료를 받던 중 보험 기간이 끝났을 때에도 퇴원하기 전까지의 계속중인 '
 '입원에 대하여는 제1항에 따라 반\n'
 '려견 위탁비용을 계속 보장합니다.\n'
 '⑧ 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회사는 반려견 위탁비용의 전부 또는 일부를 지급하지 '
 '않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 126},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000793',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
