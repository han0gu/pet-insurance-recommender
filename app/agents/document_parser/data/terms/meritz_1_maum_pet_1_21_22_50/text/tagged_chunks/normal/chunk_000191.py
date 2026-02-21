from langchain_core.documents import Document

chunk = Document(
    page_content=('변경되는 경우에는 이미 지정된 지정대리청구인의 자격은 자동적으로 상실된 것으로\n'
 '봅니다.제4조(지정대리청구인의 변경지정)① 계약자는 다음의 서류를 제출하고 지정대리청구인을 변경 지정할 수 있습니다. 이 경우\n'
 '회사는 변경 지정을 서면으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.1. 지정대리청구인 변경신청서(회사양식)\n'
 '2. 지정대리청구인의 주민등록등본, 가족관계등록부(기본증명서 등)\n'
 '3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인이 아'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000191',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
