from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 변연절제를 동반하지 않은 단순 창상봉합술\n'
 '- - 절개, 배농 또는 도관삽입술\n'
 '- - 중이내튜브유치술(중이내 환기관 삽입술)\n'
 '- - 추간판 관련 경막외 신경차단술\n'
 '- - 치, 치수, 치은, 치근, 치조골의 처치\n'
 '※ 본 시술들은 수술의 정의에 해당하지 않는 시술의 예시로, 예시에 기재되어 있지 않다 하더라도\n'
 '수술의 정의에 해당하지 않는 경우 보상되지 않습니다.# 제6조 (보험금을 지급하지 않는 사유)- ① 회사는 특별약관 일반사항 '
 '제7조(보험금을 지급하지 않는 사유)에서 정한 사유를 원인'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000330',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
