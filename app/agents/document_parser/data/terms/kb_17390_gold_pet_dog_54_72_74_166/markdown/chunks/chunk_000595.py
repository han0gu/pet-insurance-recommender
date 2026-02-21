from langchain_core.documents import Document

chunk = Document(
    page_content=('- inhibitor) 약물과 사이토포인트(Cytopoint), 트릴로스탄(Trilostane), 피모벤단\n'
 '- (Pimobendan) 또는 크리스데살라진(Crisdesalazine)(제다큐어)를 사용하여 시행\n'
 '- 한 치료를 말합니다.\n'
 '| 용 어 풀 | 이 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
