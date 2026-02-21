from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 조정에 관한 및\n'
 '13. 가입동물이 질병을 전염시켜 발생한 배상책임\n'
 '비용 질\n'
 '14. 동물보호법 시행규칙 제2조에 따른 맹견의 경우 동물보호법 제21조 제1항 2호 병\n'
 '라. 보험증권상의 보상한도액내의 금액에 대한 공탁보증보험료. 그러나 회사\n'
 '에 따라 목줄과 입마개를 하지 않아 발생한 손해에 대한 배상책임\n'
 '는 그러한 보증을 제공할 책임은 부담하지 않습니다.\n'
 '마. 피보험자가 제13조(손해배상청구에 대한 회사의 해결) 제2항 및 제3항의\n'
 '제6조(손해의 발생과 통지)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000669',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
