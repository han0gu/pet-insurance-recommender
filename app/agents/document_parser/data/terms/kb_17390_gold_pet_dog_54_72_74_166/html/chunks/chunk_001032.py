from langchain_core.documents import Document

chunk = Document(
    page_content=('함은 수의사가 반려동물의 상해 또는 질병<br>의 치료를 직접적인 목적으로 아포퀠(Apoquel) 등의 JAK '
 'inhibitor(Janus kinase<br>inhibitor) 약물과 사이토포인트(Cytopoint), '
 '트릴로스탄(Trilostane), 피모벤단<br>(Pimobendan) 또는 크리스데살라진(Crisdesalazine)(제다큐어)를 '
 "사용하여 시행<br>한 치료를 말합니다.</p><br><table id='4' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>용 어"),
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
