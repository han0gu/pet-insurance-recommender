from langchain_core.documents import Document

chunk = Document(
    page_content=('3~100% 장해지급률에 해당하는 장해상태가 되었을 때에는 장해분류표에서 정한 지급률\n'
 '을 보험증권에 기재된 이 특별약관의 보험가입금액(이하「상해 후유장해보험가입금액」\n'
 '이라 합니다)에 곱하여 산출한 금액을 상해 후유장해보험금으로 보험수익자에게 지급합\n'
 '니다.# <용어풀이># [장해지급률]질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로\n'
 '나타낸 것을 말합니다.# 제 2조 (보험금 지급에 관한 세부규정)① 제1조(보험금의 지급사유)에서 장해지급률이 상해 발생일부터 180일 '
 '이내에 확정되지'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000292',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
