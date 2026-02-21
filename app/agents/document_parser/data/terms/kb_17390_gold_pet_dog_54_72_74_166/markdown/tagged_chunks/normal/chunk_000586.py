from langchain_core.documents import Document

chunk = Document(
    page_content=('| 예 시 최대 보상한도액의 적용 특 [동일한 날에 MRI/CT 시행 후 항암약물치료 시행시 최대 보상한도액 예시] 별 '
 '·MRI/CT(100만원), 항암약물치료(30만원) 기준 약 예시① 관 ·MRI/CT 및 항암약물치료에 대한 연간 지급한도가 각각 1회 '
 '이상 남아있는 경 우 ·최대 보상한도액 = {MRI/CT 보상한도액(100만원), 항암약물치료 보상한도액 (30만원)} 중 높은 금액 = '
 '100만원 상 예시② 해 ·MRI/CT에 대한 연간 지급한도(연간 1회한)가 모두 소진된 경우 | 예 시 최대 보상한도액의 적용 특 '
 '[동일한'),
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
 'indexing': {'chunk_id': 'chunk_000586',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
