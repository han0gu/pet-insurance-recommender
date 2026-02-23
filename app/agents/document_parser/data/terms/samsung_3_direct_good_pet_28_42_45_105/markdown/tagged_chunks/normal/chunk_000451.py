from langchain_core.documents import Document

chunk = Document(
    page_content=('| 잔존유치 | 영구치가 났는데도 불구하고 유치가 남아있어서 발치를 하는 경우 |\n'
 '잠복고환고환이 음낭까지 내려오지 못하는 증상③ 제2항에서 정한 조치에 다른 진료를 병행하여 실시한 경우에는 제2항에서 정한# 조치(마취 '
 '비용을 포함합니다.)에 대한 보험금은 지급하지 않습니다.# 제 4조 (보험금의 청구)# ① 보험수익자는 다음의 서류를 제출하고 보험금을 '
 '청구하여야 합니다.- 1. 보험금 청구서(회사 양식)\n'
 '- 2. 등록견의 경우에는 동물등록증 또는 등록번호'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000451',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
