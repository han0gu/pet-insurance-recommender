from langchain_core.documents import Document

chunk = Document(
    page_content=('. 가입동물의 소음, 냄새, 털날림으로 인하여 발생한 배상책임<br>다. 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 '
 '조정에 관한 및<br>13. 가입동물이 질병을 전염시켜 발생한 배상책임<br>비용 질<br>14. 동물보호법 시행규칙 제2조에 따른 '
 '맹견의 경우 동물보호법 제21조 제1항 2호 병<br>라. 보험증권상의 보상한도액내의 금액에 대한 공탁보증보험료. 그러나 회사<br>에 '
 '따라 목줄과 입마개를 하지 않아 발생한 손해에 대한 배상책임<br>는 그러한 보증을 제공할 책임은 부담하지 않습니다.<br>마'),
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
 'indexing': {'chunk_id': 'chunk_001154',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
