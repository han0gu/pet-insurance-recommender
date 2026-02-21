from langchain_core.documents import Document

chunk = Document(
    page_content=('려야 합니다.# 제 4조 (준용규정)이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.- 138 -5-6. 지정대리청구서비스Ⅱ '
 '특별약관# 제 1조 (적용대상)이 특별약관은 보험계약자(이하「계약자」라 합니다), 피보험자 및 보험수익자가 모두 동\n'
 '일한 보험계약(특별약관이 부가된 경우에는 그 특별약관을 포함합니다. 이하「보험계약」\n'
 '이라 합니다)에 적용합니다.# 제 2조 (특별약관의 체결 및 소멸)- ① 이 특별약관은 계약자의 청약과 보험회사(이하「회사」라 합니다)의 '
 '승낙으로 보험계\n'
 '- 약에 부가하여 이루어집니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000750',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
